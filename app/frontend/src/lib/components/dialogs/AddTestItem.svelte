<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	let { test_id } = $props();

	import type { TestItem } from '$lib/types/types.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { Label } from '$lib/components/ui/label/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Textarea } from '$lib/components/ui/textarea/index.ts';

	let formItemLabel: string = $state("");
	let formItemQuestion: string = $state("");
	let formItemIsProblemSolving: boolean = $state(false);
	let formItemEARQ: string = $state("");

	async function addTestItem() {
		// TODO: finish
		try {
			const response = await fetch(
				`${apiBaseUrl}/api/test_instances/${test_id}/items`,
				{
					method: "POST",
					headers: {'Content-Type': 'application/json',},
					body: JSON.stringify({}),
				}
			);

			alert(response.status + response.statusText)
		} catch (e) {
			alert("Failed completing operation, check your connection and try again.")
		}
	}
</script>

<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>Add new test item</Dialog.Title>
	</Dialog.Header>
	<!-- TODO: finish -->
	<Label for="item_id">Item label (item_id)</Label>
	<Input id="item_id" bind:value={formItemLabel} />
	<Label for="item_id">Question</Label>
	<Textarea id="item_id" rows={4} bind:value={formItemQuestion} />
	<Label for="is_problem_solving">Type of question</Label>
	<Input id="is_probem_solving" type="radio" name="short">Short form</Input>
	<Input id="is_probem_solving" type="radio" name="problem">Problem solving</Input>
	<Label for="e_a_r_q">Expected answer</Label>
	<Textarea id="e_a_r_q" rows={6} bind:value={formItemEARQ} />
	<Dialog.Footer>
		<button class="button-primary" onclick={() => addTestItem()}>
			Add item
		</button>
	</Dialog.Footer>
</Dialog.Content>