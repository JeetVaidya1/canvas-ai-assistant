-- schema.sql — Canvas AI Assistant database (Supabase / Postgres)
-- Rebuilds everything the app expects after the DB was deleted.
--
-- Embedding dimension is 1024 to match the local BGE-large model
-- (providers/local_embeddings.py). If you ever change the embedding model,
-- update vector(1024) here AND re-ingest every course.
--
-- How to apply:
--   1. Supabase dashboard -> SQL Editor -> paste this whole file -> Run.
--   2. Use the project's *service_role* key as SUPABASE_KEY in .env
--      (RLS is left disabled below; this is a single-user personal app).

create extension if not exists vector;
create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- Core course tables
-- ---------------------------------------------------------------------------
create table if not exists courses (
    course_id   text primary key,
    title       text,
    created_at  timestamptz not null default now()
);

create table if not exists files (
    id            bigint generated always as identity primary key,
    course_id     text not null references courses(course_id) on delete cascade,
    filename      text not null,
    storage_path  text,
    file_type     text,
    ext           text,
    num_chunks    integer,
    uploaded_at   timestamptz not null default now(),
    unique (course_id, filename)            -- ingest.py upserts on_conflict="course_id,filename"
);
create index if not exists files_course_idx on files (course_id);

-- ---------------------------------------------------------------------------
-- Vector store
-- ---------------------------------------------------------------------------
create table if not exists embeddings (
    id         bigint generated always as identity primary key,
    course_id  text not null references courses(course_id) on delete cascade,
    doc_name   text,
    chunk_id   integer,
    embedding  vector(1024),
    content    text,
    page       integer,
    slide      integer,
    section    text,
    sha256     text
);
create index if not exists embeddings_course_idx on embeddings (course_id);
create index if not exists embeddings_doc_idx    on embeddings (course_id, doc_name);
-- Cosine ANN index. HNSW build may need a larger maintenance_work_mem on big
-- corpora: run `set maintenance_work_mem = '128MB';` before this if it errors.
create index if not exists embeddings_vec_idx
    on embeddings using hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Chat
-- ---------------------------------------------------------------------------
create table if not exists chat_sessions (
    id          uuid primary key default gen_random_uuid(),
    user_id     text,
    course_id   text,
    title       text,
    created_at  timestamptz not null default now()
);
create index if not exists chat_sessions_user_idx on chat_sessions (user_id);

create table if not exists messages (
    id          bigint generated always as identity primary key,
    session_id  uuid references chat_sessions(id) on delete cascade,
    role        text,
    content     text,
    sources     jsonb,          -- citation chips for assistant messages: [{file, page}]
    "timestamp" timestamptz not null default now()
);
create index if not exists messages_session_idx on messages (session_id);

-- ---------------------------------------------------------------------------
-- Exams
-- ---------------------------------------------------------------------------
create table if not exists exam_sessions (
    id               uuid primary key,            -- set explicitly in code (uuid4)
    user_id          text,
    course_id        text,
    exam_name        text,
    exam_data        jsonb,
    status           text,
    current_question integer,
    user_answers     jsonb,
    start_time       timestamptz,
    end_time         timestamptz,
    time_remaining   integer,
    is_paused        boolean default false,
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now(),
    final_score      jsonb
);
create index if not exists exam_sessions_user_idx   on exam_sessions (user_id);
create index if not exists exam_sessions_status_idx on exam_sessions (status);

-- ---------------------------------------------------------------------------
-- Learning analytics
-- ---------------------------------------------------------------------------
create table if not exists learning_progress (
    id             bigint generated always as identity primary key,
    user_id        text,
    course_id      text,
    topic          text,
    mastery_level  double precision,
    last_reviewed  timestamptz,
    review_count   integer default 0,
    unique (user_id, course_id, topic)
);

create table if not exists user_interactions (
    id               bigint generated always as identity primary key,
    user_id          text,
    course_id        text,
    question         text,
    answer           text,
    confidence_score double precision,
    response_time    double precision,
    question_type    text,
    "timestamp"      timestamptz not null default now()
);
create index if not exists user_interactions_idx on user_interactions (user_id, course_id);

-- ---------------------------------------------------------------------------
-- Notes
-- ---------------------------------------------------------------------------
create table if not exists notes (
    id           uuid primary key default gen_random_uuid(),  -- code may set uuid4
    course_id    text,
    title        text,
    content      text,
    source_files jsonb,
    topic_focus  text,
    topics       jsonb,
    word_count   integer,
    reading_time text,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);
create index if not exists notes_course_idx on notes (course_id);

-- ---------------------------------------------------------------------------
-- Past papers
-- ---------------------------------------------------------------------------
create table if not exists past_papers (
    id            uuid primary key default gen_random_uuid(),
    course_id     text,
    filename      text,
    storage_path  text,
    analysis_data jsonb,
    uploaded_by   text,
    uploaded_at   timestamptz not null default now()
);
create index if not exists past_papers_course_idx on past_papers (course_id);

create table if not exists past_paper_analyses (
    id            bigint generated always as identity primary key,
    course_id     text,
    filename      text,
    analysis_data jsonb,
    created_at    timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- RPC functions used by vector_store.py (1024-dim cosine search)
-- ---------------------------------------------------------------------------
create or replace function match_embeddings_enhanced(
    query_embedding vector(1024),
    course_id_param text,
    match_threshold double precision default 0.1,
    match_count     integer default 5
)
returns table (
    content    text,
    doc_name   text,
    chunk_id   integer,
    similarity double precision,
    course_id  text,
    page       integer,
    slide      integer,
    section    text,
    sha256     text
)
language sql stable
as $$
    select e.content, e.doc_name, e.chunk_id,
           1 - (e.embedding <=> query_embedding) as similarity,
           e.course_id, e.page, e.slide, e.section, e.sha256
    from embeddings e
    where e.course_id = course_id_param
      and e.embedding is not null
      and 1 - (e.embedding <=> query_embedding) >= match_threshold
    order by e.embedding <=> query_embedding
    limit match_count;
$$;

create or replace function match_embeddings(
    query_embedding vector(1024),
    course_id_param text,
    match_count     integer default 5
)
returns table (
    content    text,
    doc_name   text,
    chunk_id   integer,
    similarity double precision,
    course_id  text,
    page       integer,
    slide      integer,
    section    text,
    sha256     text
)
language sql stable
as $$
    select e.content, e.doc_name, e.chunk_id,
           1 - (e.embedding <=> query_embedding) as similarity,
           e.course_id, e.page, e.slide, e.section, e.sha256
    from embeddings e
    where e.course_id = course_id_param
      and e.embedding is not null
    order by e.embedding <=> query_embedding
    limit match_count;
$$;

create or replace function match_embeddings_by_document(
    query_embedding vector(1024),
    course_id_param text,
    doc_name_param  text,
    match_count     integer default 5
)
returns table (
    content    text,
    doc_name   text,
    chunk_id   integer,
    similarity double precision,
    course_id  text,
    page       integer,
    slide      integer,
    section    text,
    sha256     text
)
language sql stable
as $$
    select e.content, e.doc_name, e.chunk_id,
           1 - (e.embedding <=> query_embedding) as similarity,
           e.course_id, e.page, e.slide, e.section, e.sha256
    from embeddings e
    where e.course_id = course_id_param
      and e.doc_name  = doc_name_param
      and e.embedding is not null
    order by e.embedding <=> query_embedding
    limit match_count;
$$;

-- Keyword (full-text) search for hybrid retrieval, fused with vector search
-- via reciprocal rank fusion in rag/retrieval.py.
create index if not exists embeddings_fts_idx
    on embeddings using gin (to_tsvector('english', coalesce(content, '')));

create or replace function keyword_search_embeddings(
    course_id_param text,
    query_text      text,
    match_count     integer default 20
)
returns table (
    content   text,
    doc_name  text,
    chunk_id  integer,
    course_id text,
    page      integer,
    slide     integer,
    section   text,
    sha256    text,
    rank      real
)
language sql stable
as $$
    select e.content, e.doc_name, e.chunk_id, e.course_id,
           e.page, e.slide, e.section, e.sha256,
           ts_rank(to_tsvector('english', coalesce(e.content, '')),
                   websearch_to_tsquery('english', query_text)) as rank
    from embeddings e
    where e.course_id = course_id_param
      and to_tsvector('english', coalesce(e.content, '')) @@ websearch_to_tsquery('english', query_text)
    order by rank desc
    limit match_count;
$$;

-- ---------------------------------------------------------------------------
-- Storage bucket (public so storage.py's public URLs resolve)
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('course-files', 'course-files', true)
on conflict (id) do nothing;
