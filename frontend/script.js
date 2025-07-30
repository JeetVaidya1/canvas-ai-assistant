const API_BASE = "http://127.0.0.1:8000";

async function createCourse() {
  const id = document.getElementById("course-id").value.trim();
  const title = document.getElementById("course-title").value.trim();
  if (!id || !title) return alert("Please enter both course ID and title.");

  const form = new FormData();
  form.append("course_id", id);
  form.append("title", title);

  const res = await fetch(`${API_BASE}/create-course`, {
    method: "POST",
    body: form,
  });

  if (res.ok) {
    alert(`✅ Course ${id} created!`);
    document.getElementById("course-id").value = "";
    document.getElementById("course-title").value = "";
    await loadCourses(); // 🔁 force reload everything
  } else {
    alert("❌ Error creating course");
  }
}

async function loadCourses() {
  const res = await fetch(`${API_BASE}/list-courses`);
  const courses = await res.json();
  const container = document.getElementById("course-list");
  const dropdown = document.getElementById("upload-course-id");

  container.innerHTML = "";
  dropdown.innerHTML = "<option disabled selected>Select a course</option>";

  courses.forEach((course) => {
    // UI block
    const div = document.createElement("div");
    div.innerHTML = `
      <strong>${course.course_id}:</strong> ${course.title}
      <button onclick="viewFiles('${course.course_id}')">📂 View Files</button>
      <button onclick="deleteCourse('${course.course_id}')">🗑 Delete Course</button>
      <div id="files-${course.course_id}"></div>
    `;
    container.appendChild(div);

    // Upload dropdown
    const option = document.createElement("option");
    option.value = course.course_id;
    option.textContent = `${course.course_id} - ${course.title}`;
    dropdown.appendChild(option);
  });
}

async function uploadFiles() {
  const courseId = document.getElementById("upload-course-id").value;
  const input = document.getElementById("file-input");
  const files = input.files;
  if (!courseId || files.length === 0) return alert("Select course and files");

  const form = new FormData();
  for (let file of files) {
    form.append("files", file);
  }
  form.append("course_id", courseId);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: form,
  });

  if (res.ok) {
    alert("✅ Files uploaded!");
    input.value = null;
    viewFiles(courseId);
  } else {
    alert("❌ Upload failed");
  }
}

async function viewFiles(courseId) {
  const res = await fetch(`${API_BASE}/list-files?course_id=${courseId}`);
  const files = await res.json();
  const fileDiv = document.getElementById(`files-${courseId}`);
  if (!Array.isArray(files)) return;

  fileDiv.innerHTML = "<ul>" + files.map(file =>
    `<li>${file}
      <button onclick="deleteFile('${courseId}', '${file}')">🗑 Delete</button>
    </li>`).join("") + "</ul>";
}

async function deleteCourse(courseId) {
  const confirmed = confirm(`Are you sure you want to delete course "${courseId}"?`);
  if (!confirmed) return;

  const res = await fetch(`${API_BASE}/delete-course/${courseId}`, {
    method: "DELETE",
  });
  if (res.ok) {
    alert("✅ Course deleted");
    await loadCourses();
  } else {
    alert("❌ Failed to delete course");
  }
}

async function deleteFile(courseId, filename) {
  const confirmed = confirm(`Delete file "${filename}" from "${courseId}"?`);
  if (!confirmed) return;

  const res = await fetch(`${API_BASE}/delete-file/${courseId}/${filename}`, {
    method: "DELETE",
  });
  if (res.ok) {
    alert("✅ File deleted");
    viewFiles(courseId);
  } else {
    alert("❌ Failed to delete file");
  }
}

// 🚀 Initial render
window.onload = loadCourses;
