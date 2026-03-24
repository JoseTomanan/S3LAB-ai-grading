import { API_URL } from '$lib/constants.ts';
import type { PageLoad } from "./$types.js";
import type { GetSpecificEvaluationResponse, StudentAnswer, TestItem } from "$lib/index.ts";
import { REPOPULATE_UNANSWERED_ITEMS } from "$lib/utils/ai_evaluations.ts";
import { error } from "@sveltejs/kit";



export const load: PageLoad = async ({ fetch, params, parent }) => {
  const { test_items } = await parent();

  if (typeof test_items === "undefined")
    throw new Error("test_items is undefined");

  let student_items: (StudentAnswer & {label: string})[] | undefined = undefined;
  let student_ai_evaluations: GetSpecificEvaluationResponse[] = [];
  REPOPULATE_UNANSWERED_ITEMS(student_ai_evaluations, test_items);

  try {
    const responseOne = await fetch(`${API_URL}/api/student_answers/${params.test_id}/${params.student_no}`);
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
    console.log("Failed to fetch student answers:\n"+e);
    throw error(500, "Failed to fetch student answers.");
  }
  
  try {
    const response = await fetch(`${API_URL}/api/student_answers/${params.test_id}/results/${params.student_no}`);
    switch (response.status) {
    case 200:
      const result = await response.json();
      student_ai_evaluations = result.evaluations ?? [];
      REPOPULATE_UNANSWERED_ITEMS(student_ai_evaluations, test_items);
      break;
    default:
      console.error(`${response.status} ${response.statusText}`);
    }
  } catch (e) {
    console.log("Failed to fetch AI evaluations:\n"+e)
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

  return { student_items, student_ai_evaluations };
};
