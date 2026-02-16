<script lang="ts">
	import { API_BASE_URL } from '$lib/constants.ts';
	
	import MdiDelete from "~icons/mdi/delete";

	import type { TestItem } from '$lib/index.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { Button } from '$lib/components/ui/button/index.ts';
	import { Label } from '$lib/components/ui/label/index.js';
	import { Textarea } from '$lib/components/ui/textarea/index.js';
	import { Input } from '$lib/components/ui/input/index.ts';

	let { testItem, test_id } = $props<{
		testItem: TestItem,
		test_id: string
	}>();

	let formTestItem: TestItem = $state({
				item_id: testItem.item_id,
				label: testItem.label,
				question: testItem.question,
				is_problem_solving: testItem.is_problem_solving,
				expected_answer_rubric_questions: testItem.expected_answer_rubric_questions,
			});

	async function editTestItem(submittedTestItem: TestItem) {
		if (testItem != submittedTestItem) {
			try {
				const formBody = {
							label: submittedTestItem.label,
							question: submittedTestItem.question,
							expected_answer_rubric_questions: submittedTestItem.expected_answer_rubric_questions,
							};

				console.log(formBody);
				
				const response = await fetch(
							`${API_BASE_URL}/api/test_instances/${test_id}/items/${submittedTestItem.item_id}`,
							{
								method: "PATCH",
								headers: {'Content-Type': 'application/json',},
								body: JSON.stringify(formBody),
							}
							);
				
				if (response.ok) {
					const result = await response.json();
					alert("Success: " + result.item_id);
					window.location.reload();
				} else
					alert(response.status + response.statusText);
			} catch (e) {
				alert("Failed to send POST request.");
			}
		} else
			alert("No changes were made.");
	}

	async function deleteTestItem() {
		try {
			const response = await fetch(
					`${API_BASE_URL}/api/test_instances/${test_id}/${formTestItem.item_id}`,
					{
						method: "DELETE",
						headers: {'Content-Type': 'application/json',},
					}
				);
				
				if (response.ok) {
					alert("Delete success.");
					window.location.reload();
				} else
					alert(response.status + response.statusText);
		} catch (e) {
			alert("Failed; try again.")
		}
	}
</script>


<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>
			Edit test item {testItem.label}
		</Dialog.Title>
	</Dialog.Header>
	<Label for="label">Label</Label>
	<Input id="label" bind:value={ formTestItem.label } required/>
	<Label for="question">Question</Label>
	<Textarea id="question" rows={4} bind:value={ formTestItem.question } required/>
	<Label for="e_a_r_q">
		{ testItem.is_problem_solving
					? "Rubric questions (separate with `; `)"
					: "Expected answer"}
	</Label>
	<Textarea id="e_a_r_q" rows={6} bind:value={ formTestItem.expected_answer_rubric_questions } required/>
	<Dialog.Footer>
		<div class="flex flex-row w-full gap-2">
			<Button variant="outline"
							class="w-4/5"
							onclick={() => editTestItem(formTestItem)}>
				Save changes
			</Button>
			<Button variant="destructive"
							class="w-1/5"
							onclick={() => deleteTestItem()}>
				<MdiDelete class="size-6"/>
			</Button>
		</div>
	</Dialog.Footer>
</Dialog.Content>