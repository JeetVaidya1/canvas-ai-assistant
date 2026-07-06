/** Basename of a source path — `slides/week3.pdf` → `week3.pdf`. */
export function fileLabel(file: string): string {
  const parts = file.split(/[/\\]/)
  return parts[parts.length - 1] || file
}
