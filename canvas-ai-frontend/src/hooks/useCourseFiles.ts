import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  listFiles,
  uploadFiles,
  deleteFile,
} from '@/lib/api'

export function useCourseFiles(courseId: string | undefined) {
  return useQuery({
    queryKey: ['files', courseId],
    queryFn: () => listFiles(courseId!),
    enabled: !!courseId,
  })
}

export function useUploadFile(courseId: string | undefined) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (files: File[]) => uploadFiles(courseId!, files),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['files', courseId] }),
  })
}

export function useDeleteFile(courseId: string | undefined) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (filename: string) => deleteFile(courseId!, filename),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['files', courseId] }),
  })
}
