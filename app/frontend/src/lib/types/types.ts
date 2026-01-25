// TODO: segregate into different files (soon)

export enum RenderStatus {
	INCOMPLETE = "Incomplete",
	IN_PROGRESS = "In progress",
	DONE = "Done",
}

export interface TestInstance {
	test_name: string;
	section: string;
	date: string;
	is_done_rendering: boolean;
}

export interface TestQuestion {
	number: number,
	question: string;
	is_problem_solving: boolean;
	expected_answer_rubric_question: string;	// if rubric question? separate with `. `
}
