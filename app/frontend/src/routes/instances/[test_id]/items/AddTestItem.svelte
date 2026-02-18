<script lang="ts">
	const { test_id } = $props();
	
	import { API_BASE_URL } from '$lib/constants.ts';
	import MdiPlus from "~icons/mdi/plus";

	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import * as RadioGroup from '$lib/components/ui/radio-group/index.ts';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Label } from '$lib/components/ui/label/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Textarea } from '$lib/components/ui/textarea/index.ts';

	let formItemLabel: string = $state("");
	let formItemQuestion: string = $state("");
	let formItemIsProblemSolving: boolean = $state(false);
	let formItemRQ: {question: string, points: number}[] = $state([{question: "", points: 0}]);
	let formItemEA: {question: string, points: number} = $state({question: "", points: 0});
	
	let submittableEARQ: string = "";

	async function addTestItem() {
		if (formItemIsProblemSolving) {
			for (const item of formItemRQ) {
				if (item.points != 0)
					submittableEARQ = submittableEARQ.concat(`${item.question} [${item.points}pts];`);
			}
		} else {
			submittableEARQ = `${formItemEA.question} [${formItemEA.points}pts]`
		}

		try {
			const response = await fetch(
				`${API_BASE_URL}/api/test_instances/${test_id}/items`,
				{
					method: "POST",
					headers: {'Content-Type': 'application/json',},
					body: JSON.stringify({
						label: formItemLabel,
						question: formItemQuestion,
						is_problem_solving: formItemIsProblemSolving,
						expected_answer_rubric_questions: submittableEARQ,
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
	{#if formItemIsProblemSolving}
		<Label for="r_q"
						class="flex flex-row justify-between">
			Rubric questions
			<button class="px-1 py-0 m-0 bg-secondary"
							onclick={() => {formItemRQ.push({question: "", points: 0})}}>
				<MdiPlus class="size-3"/>
			</button>
		</Label>
		<div class="space-y-1">
			{#each formItemRQ as item}
				<span class="flex flex-row gap-1">
					<Input id="r_q"
									class="w-5/6"
									pattern="[^();]*"
									bind:value={item.question} />
					<Input id="r_q" class="w-1/6"
									type="number"
									bind:value={item.points} />
				</span>
			{/each}
		</div>
	{:else}
		<Label for="e_a">Expected answer</Label>
		<span class="flex flex-row gap-1">
			<Input id="e_a"
							class="w-5/6"
							pattern="[^();]*"
							bind:value={formItemEA.question} />
			<Input id="e_a"
							class="w-1/6"
							type="number"
							bind:value={formItemEA.points} />
		</span>
	{/if}
	<Dialog.Footer>
		<Button variant="outline"
						onclick={() => addTestItem()}>
			Add item
		</Button>
	</Dialog.Footer>
</Dialog.Content>