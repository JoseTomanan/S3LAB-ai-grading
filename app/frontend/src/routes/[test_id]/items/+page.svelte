<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	let { data } = $props();

	import type { PageData } from './$types.d.ts';
	import { onMount } from 'svelte';
	import MdiEditOutline from '~icons/mdi/edit-outline';
	import MdiPlus from '~icons/mdi/plus';
	
	import type { TestInstance, TestItem } from '$lib/types/types.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import EditTestItem from '$lib/components/dialogs/EditTestItem.svelte';
	import AddTestItem from '$lib/components/dialogs/AddTestItem.svelte';

	let isPageLoading: boolean = $state(true);

	let allItems: TestItem[] = $state([]);

	let shortFormItems: TestItem[] = $derived(allItems.filter(item => !item.is_problem_solving));
	let probSolItems: TestItem[] = $derived(allItems.filter(item => item.is_problem_solving));
	
	onMount(async () => {
		try {
			const response = await fetch(
				`${apiBaseUrl}/api/test_instances/${data.test_id}/items`,
				{
					method: "GET",
					headers: {'Content-Type': 'application/json',},
					body: JSON.stringify({}),
				}
			);

			const result = await response.json();
			allItems = result.items;
		} catch (e) {
			// FIXME: revert to alert() once functional
			console.log("Failed to fetch test details:\n"+e);
		} finally {
			isPageLoading = false;
		}
	});
	
	// FIXME: remove once API is working
	setTimeout(() => {
		allItems = [
			{item_id: '1', question: "David and Goliath divide a pie in half. If they were to divide it evenly, how many should each one get?", is_problem_solving: false, expected_answer_rubric_questions: ""},
			{item_id: '2', question: "Three people are to share a pie evenly. Using a circle, illustrate how this pie will be sliced.", is_problem_solving: true, expected_answer_rubric_questions: ""},
		];

		isPageLoading = false;
	}, 1000);

</script>

<div class="px-4 space-y-4">
	{#if isPageLoading}
		<p>Loading...</p>
	{:else}
		{#each [{a: "Short Form Items", b: shortFormItems}, {a: "Problem-Solving Items", b: probSolItems}] as bigItem}
			<div class="card p-2">
				<span class="flex flex-row items-center w-full justify-between">
					<h3>{ bigItem.a }</h3>
					<Dialog.Root>
						<Dialog.Trigger>
							<MdiPlus class="size-8"/>
						</Dialog.Trigger>
						<AddTestItem />
					</Dialog.Root>
				</span>
				<div class="ml-4">
					{#each bigItem.b as smallItem}
						<span class="flex flex-row items-end justify-between">
							<p class="truncate text-ellipsis w-fill">
								({smallItem.item_id}) {smallItem.question}
							</p>
							<Dialog.Root>
								<Dialog.Trigger>
									<MdiEditOutline class="size-4"/>
								</Dialog.Trigger>
								<EditTestItem testItem={smallItem} test_id={data.test_id}/>
							</Dialog.Root>
						</span>
					{/each}
				</div>
			</div>
		{/each}
	{/if}
</div>