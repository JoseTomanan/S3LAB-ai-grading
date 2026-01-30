// TODO: segregate into different files (soon)

export enum RenderStatus {
	INCOMPLETE = "Incomplete",
	IN_PROGRESS = "In progress",
	DONE = "Done",
}

export interface TestInstance {
	name: string;
	section: string;
	date: string;
	test_id: string;
	is_done_rendering: boolean;
}

export interface TestItem {
	item_id: string;
	question: string;
	is_problem_solving: boolean;
	expected_answer_rubric_questions: string;	// if rubric question, separate with `. `
}
