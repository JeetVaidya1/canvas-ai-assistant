import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  fetchCourses,
  createCourse,
  deleteCourse,
} from '@/lib/api'

export function useCourses() {
  return useQuery({
    queryKey: ['courses'],
    queryFn: fetchCourses,
  })
}

export function useCreateCourse() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ courseId, title }: { courseId: string; title: string }) =>
      createCourse(courseId, title),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['courses'] }),
  })
}

export function useDeleteCourse() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (courseId: string) => deleteCourse(courseId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['courses'] }),
  })
}
