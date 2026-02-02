export interface TestInstance {
	test_id: string;
	name: string;
	section: string;
	date: string;
	is_done_rendering: boolean;
}


export interface TestItem {
	item_id: number;
	label: string;
	question: string;
	is_problem_solving: boolean;
	expected_answer_rubric_questions: string;
}


export interface TestItemHeader {
	item_id: number;
	label: string;
}


export interface TestPaper {
	paper_id: number;
	student_no: string;
	test_id: string;
	is_done_rendering: boolean;
}


export interface Student {
	student_no: string;
	name: string;
	section: string;
}


export interface StudentAnswer {
	answer_id: string;
	student_no: string;
	item_id: string;
	image_directory: string;
	ai_evaluation: string;
	is_done_rendering: boolean;
}


export interface Section {
	section_id: number;
	section: string;
}
