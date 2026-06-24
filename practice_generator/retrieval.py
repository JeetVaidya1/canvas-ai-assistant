"""Retrieval/context mixin for practice generation.

Builds smart search queries, performs last-resort content searches,
deduplicates results, and assembles subject-agnostic context strings used
to ground generated practice problems in course material.
"""
from typing import List, Dict


class RetrievalMixin:
    """Search-query construction, fallback search, and context assembly."""

    def create_smart_search_queries(self, topic: str) -> List[str]:
        """Create smarter search queries based on the topic"""
        queries = []

        # Original topic
        queries.append(topic)

        # Clean up the topic for better matching
        topic_clean = topic.lower().strip()

        # Handle specific cases
        if "bst" in topic_clean:
            queries.extend([
                "binary search tree",
                "BST operations",
                "tree insertion deletion",
                "binary tree search"
            ])
        elif "stack" in topic_clean:
            queries.extend([
                "stack data structure",
                "stack operations",
                "push pop operations",
                "LIFO structure"
            ])
        elif "tree" in topic_clean and "part" in topic_clean:
            queries.extend([
                "binary tree",
                "tree traversal",
                "tree terminology",
                "tree structure"
            ])
        elif "graph" in topic_clean:
            queries.extend([
                "graph data structure",
                "graph algorithms",
                "adjacency matrix",
                "graph traversal"
            ])
        elif "heap" in topic_clean or "priority queue" in topic_clean:
            queries.extend([
                "heap data structure",
                "priority queue",
                "binary heap",
                "heap operations"
            ])
        elif "hash" in topic_clean:
            queries.extend([
                "hash table",
                "hashing function",
                "collision resolution",
                "hash map"
            ])
        elif "sort" in topic_clean:
            queries.extend([
                "sorting algorithms",
                "merge sort",
                "quick sort",
                "sorting comparison"
            ])
        elif "algorithm" in topic_clean and "analysis" in topic_clean:
            queries.extend([
                "time complexity",
                "space complexity",
                "big O notation",
                "algorithm efficiency"
            ])
        elif "array" in topic_clean or "linked" in topic_clean:
            queries.extend([
                "array operations",
                "linked list",
                "list implementation",
                "dynamic array"
            ])
        else:
            # Generic fallbacks
            queries.extend([
                f"{topic} implementation",
                f"{topic} operations",
                f"{topic} algorithm"
            ])

        # Remove duplicates while preserving order
        unique_queries = []
        for q in queries:
            if q not in unique_queries:
                unique_queries.append(q)

        return unique_queries[:6]  # Limit to 6 queries

    def fallback_content_search(self, course_id: str, topic: str, vector_store) -> List[Dict]:
        """Last resort: try to find ANY content that might be relevant"""
        try:
            print(f"🔄 Trying fallback content search for: '{topic}'")

            # Extract key words from the topic
            topic_words = topic.lower().split()
            fallback_queries = []

            # Try individual words
            for word in topic_words:
                if len(word) > 3:  # Skip short words
                    fallback_queries.append(word)

            # Try common variations
            if "bst" in topic.lower():
                fallback_queries.extend(["tree", "binary", "search"])
            elif "stack" in topic.lower():
                fallback_queries.extend(["stack", "push", "pop"])

            all_results = []
            for query in fallback_queries[:4]:  # Limit fallback searches
                try:
                    print(f"  🔍 Fallback search: '{query}'")
                    emb_resp = self.openai_client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=[query]
                    )
                    results = vector_store.query(course_id, emb_resp.data[0].embedding, top_k=5)
                    if results:
                        print(f"    ✅ Fallback found {len(results)} results")
                        all_results.extend(results)
                except Exception as e:
                    print(f"    ❌ Fallback search failed: {e}")

            return self.deduplicate_results(all_results)

        except Exception as e:
            print(f"❌ Fallback search failed: {e}")
            return []

    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate content from search results"""
        seen_content = set()
        unique_results = []

        for result in results:
            content = result.get("content", "").strip()
            content_hash = hash(content[:200])  # Use first 200 chars as signature

            if content and content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
                if len(unique_results) >= 8:  # Reasonable limit
                    break

        return unique_results

    def create_universal_context(self, results: List[Dict], topic: str) -> str:
        """Create context that works for any subject - IMPROVED"""
        if not results:
            return f"No specific course materials found for {topic}."

        context_parts = [f"COURSE MATERIALS RELATED TO {topic.upper()}:"]

        for i, result in enumerate(results[:8], 1):  # Use more results
            content = result.get("content", "").strip()
            doc = result.get("doc_name", "unknown")
            page = result.get("page")

            source_info = f"[Source {i}: {doc}"
            if page:
                source_info += f", page {page}"
            source_info += "]"

            context_parts.append(f"\n{source_info}")
            context_parts.append(content)
            context_parts.append("---")

        full_context = "\n".join(context_parts)

        # If context is very short, mention that
        if len(full_context) < 500:
            context_parts.insert(1, f"\nNote: Limited course material found for {topic}. Using available related content:")

        return "\n".join(context_parts)
