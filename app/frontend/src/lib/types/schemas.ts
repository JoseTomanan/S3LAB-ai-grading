export interface EvaluationsResponse {
	item_id: number;
	label: string;
	question: string;
	expected_answer_rubric_questions: string;
	ai_evaluation: string;
}

export interface StudentStoresResponse {
	student_no: string;
	name: string;
	evaluations: EvaluationsResponse[];
}