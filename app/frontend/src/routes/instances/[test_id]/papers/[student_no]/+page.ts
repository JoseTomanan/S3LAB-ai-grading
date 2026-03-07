import type { PageLoad } from "./$types.js";
import { API_BASE_URL } from '$lib/constants.ts';
import type { StudentAnswer } from "$lib/index.ts";



export const load: PageLoad = async ({ fetch, params, parent }) => {
  const { test_items } = await parent();

  let student_items: (StudentAnswer & {label: string})[] | undefined = undefined;

  try {
    const responseOne = await fetch(
          `${API_BASE_URL}/api/student_answers/${params.test_id}/${params.student_no}`,
          {
            method: "GET",
            headers: {'Content-Type': 'application/json',},
          }
          );
    
    switch (responseOne.status) {
      case 200:
        const result = await responseOne.json();
        student_items = result ? result : [];
        break;
      case 204:
        student_items = [];
        break;
      default:
        console.log(`CALL 1: ${responseOne.status} ${responseOne.statusText}`);
    }
  } catch (e) {
    console.log("Failed to fetch student answers:\n"+e)
  }

  if (test_items && student_items) {
    for (const testItem of test_items) {
      /** For each test item, if the student does not have an answer entry for it yet, 
       *  Add a default (empty) StudentAnswer object for that test item.
       */
      if (!student_items.find(si => si.item_id === testItem.item_id))
        student_items.push({
              answer_id: 0,
              item_id: testItem.item_id,
              label: testItem.label,
              student_no: params.student_no,
              image_directory: "",
              ai_evaluation: "",
              is_done_rendering: false,
        });
    }
  }

  return { student_items };
};
