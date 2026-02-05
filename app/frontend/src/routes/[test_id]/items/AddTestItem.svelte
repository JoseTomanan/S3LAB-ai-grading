<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	let { test_id } = $props();

	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import * as RadioGroup from '$lib/components/ui/radio-group/index.ts';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Label } from '$lib/components/ui/label/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Textarea } from '$lib/components/ui/textarea/index.ts';

	let formItemLabel: string = $state("");
	let formItemQuestion: string = $state("");
	let formItemIsProblemSolving: boolean = $state(false);
	let formItemEARQ: string = $state("");

	async function addTestItem() {
		// FIXME: remove this once verified that functional
		try {
			const response = await fetch(
				`${apiBaseUrl}/api/test_instances/${test_id}/items`,
				{
					method: "POST",
					headers: {'Content-Type': 'application/json',},
					body: JSON.stringify({
						label: formItemLabel,
						question: formItemQuestion,
						is_problem_solving: formItemIsProblemSolving,
						expected_answer_rubric_questions: formItemEARQ,
					}),
				}
			);

			if (response.ok) {
				const data = await response.json();
				alert("Success:" + data.items);
			}
			else
				alert(response.status + response.statusText);
		} catch (e) {
			alert("Failed completing operation, check your connection and try again.")
		}
	}
</script>

<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>Add new test item</Dialog.Title>
	</Dialog.Header>
	<Label for="item_id">Item label</Label>
	<Input id="item_id" bind:value={formItemLabel} />
	<Label for="item_id">Question</Label>
	<Textarea id="item_id" rows={4} bind:value={formItemQuestion} />
	<Label>Type of question</Label>
	<RadioGroup.Root value="short_form"
									class="flex flex-row justify-between"
									onValueChange={(v) => formItemIsProblemSolving = (v == "prob_sol")}>
		<span class="flex flex-row gap-2 w-2/5">
			<RadioGroup.Item value="short_form" id="short_form"/>
			<Label for="short_form" class="font-normal">Short form</Label>
		</span>
		<span class="flex flex-row gap-2 w-3/5">
			<RadioGroup.Item value="prob_sol" id="prob_sol"/>
			<Label for="prob_sol" class="font-normal">Problem solving</Label>
		</span>
	</RadioGroup.Root>
	<Label for="e_a_r_q">
		{formItemIsProblemSolving ? "Rubric questions (separate with `. `)" : "Expected answer"}
	</Label>
	<Textarea id="e_a_r_q" rows={6} bind:value={formItemEARQ} />
	<Dialog.Footer>
		<Button variant="secondary" onclick={() => addTestItem()}>
			Add item
		</Button>
	</Dialog.Footer>
</Dialog.Content>