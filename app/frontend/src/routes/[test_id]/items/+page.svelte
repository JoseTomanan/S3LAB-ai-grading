<script lang="ts">
	const { data } = $props();
	
	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from 'svelte';
	
	import MdiEditOutline from '~icons/mdi/edit-outline';
	import MdiPlus from '~icons/mdi/plus';
	
	import type { TestInstance, TestItem } from '$lib/index.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import EditTestItem from './EditTestItem.svelte';
	import AddTestItem from './AddTestItem.svelte';

	let isPageLoading: boolean = $state(true);

	let allItems: TestItem[] = $state([]);

	let shortFormItems: TestItem[] = $derived(allItems.filter(item => !item.is_problem_solving));
	let probSolItems: TestItem[] = $derived(allItems.filter(item => item.is_problem_solving));
	
	onMount(async () => {
		try {
			const response = await fetch(
						`${API_BASE_URL}/api/test_instances/${data.test_id}/items`,
						{
							method: "GET",
							headers: {'Content-Type': 'application/json',},
						}
					);

			const result = await response.json();
			
			if (!response.ok)
				alert(`${response.status} ${response.statusText}`);
			else
				allItems = result.items;
		} catch (e) {
			alert("Failed to fetch, check your network connection and try again.\nERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	});
</script>


<div class="space-y-4">
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
						<AddTestItem test_id={data.test_id} />
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
					{#if bigItem.b.length == 0}
						<p class="italic">Nothing to see here. If this is a mistake, check your network connection.</p>
					{/if}
				</div>
			</div>
		{/each}
	{/if}
</div>