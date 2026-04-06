import { API_URL } from '$lib/constants.ts';
import type { LayoutLoad } from './$types.d.ts';
import type { Student } from '$lib/index.ts';
import { error } from '@sveltejs/kit';

export const load: LayoutLoad = async ({ fetch, params }) => {
  const { section_id } = params;

  try {
    const response = await fetch(`${API_URL}/api/sections/${section_id}`);

    if (!response.ok) {
      throw error(500, `Failed to fetch students for section ${section_id}`);
    }

    const students: Student[] = await response.json();
    const sortedStudents = students.toSorted((a, b) => a.student_no.localeCompare(b.student_no));

    return { section_id, students: sortedStudents };
  } catch (e) {
    console.log(`Failed to fetch students for section ${section_id}.\n` + String(e));
    throw error(500, `Failed to fetch students for section ${section_id}`);
  }
};
