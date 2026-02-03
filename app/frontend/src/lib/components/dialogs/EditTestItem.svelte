<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	import type { TestItem } from '$lib/types/types.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { Label } from '$lib/components/ui/label/index.js';
	import { Textarea } from '$lib/components/ui/textarea/index.js';
	import { Input } from '$lib/components/ui/input/index.ts';

	let { testItem, test_id } = $props<{
		testItem: TestItem,
		test_id: string
	}>();

	let formTestItem: TestItem = {
		item_id: testItem.item_id,
		label: testItem.label,
		question: testItem.question,
		is_problem_solving: testItem.is_problem_solving,
		expected_answer_rubric_questions: testItem.expected_answer_rubric_questions,
	};

	// TODO: validate workingness with backend API
	async function editTestItem(submittedTestItem: TestItem) {
		if (testItem != submittedTestItem) {
			try {
				const response = await fetch(
					`${apiBaseUrl}/api/test_instances/${test_id}/${submittedTestItem.item_id}`,
					{
						method: "PATCH",
						headers: {'Content-Type': 'application/json',},
						body: JSON.stringify({
							"label": submittedTestItem.label,
							"question": submittedTestItem.question,
							"is_problem_solving": submittedTestItem.is_problem_solving,
							"expected_answer_rubric_questions": submittedTestItem.expected_answer_rubric_questions,
						}),
					}
				);
				
				const status: number = await response.status;

				switch (status) {
					case 200:
						alert("Change successful.");
						break;
					
					case 404:
						alert("Received 404. Please check the test_id and item_id, and try again.");
						break;

					default:
						alert("Status code returned: " + status);
						break;
				}
			} catch (e) {
				alert("Failed to send POST request.");
			} finally {
				window.location.reload();
			}
		} else
			alert("No changes were made.");
	}
</script>

<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>Edit test item {testItem.item_id}</Dialog.Title>
	</Dialog.Header>
	<Label for="label">Label</Label>
	<Input id="label" bind:value={ formTestItem.label }/>
	<Label for="question">Question</Label>
	<Textarea id="question" rows={4} bind:value={ formTestItem.question } required/>
	<Label for="e_a_r_q">
		{ testItem.is_problem_solving
					? "Rubric questions (separate with `. `)"
					: "Expected answer"}
	</Label>
	<Textarea id="e_a_r_q" rows={6} bind:value={ formTestItem.expected_answer_rubric_questions } required/>
	<Dialog.Footer>
		<button class="button-primary" onclick={() => editTestItem(formTestItem)}>
			Save changes
		</button>
	</Dialog.Footer>
</Dialog.Content>