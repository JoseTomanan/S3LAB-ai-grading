import type { GetSpecificEvaluationResponse } from "$lib/types/schemas.ts";
import type { TestItem } from "$lib/types/types.ts";



export const GET_E_A_R_Q = (i: any) => i.expected_answer_rubric_questions.split(';').map((s: string) => s.trim()).filter(Boolean);
export const GET_SCORES = (i: any) => i.scores.split(';').map((s: string) => s.trim()).filter(Boolean);

export const REPOPULATE_UNANSWERED_ITEMS = (questionItemEvals: GetSpecificEvaluationResponse[], testItems: TestItem[]) => {
  const evalItemIds = new Set(questionItemEvals.map(e => e.item_id));
  const unansweredItems = testItems
          .filter(item => !evalItemIds.has(item.item_id))
          .map(item => ({
            answer_id: -1,
            item_id: item.item_id,
            label: item.label,
            question: item.question,
            expected_answer_rubric_questions: item.expected_answer_rubric_questions,
            ai_evaluation: "",
            scores: "",
          }));
  questionItemEvals.push(...unansweredItems);
};