<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	let { data, children }: {data: LayoutData, children: Snippet} = $props();

	import type { LayoutData } from './$types.d.ts';
	import type { Snippet } from 'svelte';
	import { onMount } from 'svelte';

	import MdiArrowBack from '~icons/mdi/arrow-back';
	import MdiCheckboxBlankOutline from '~icons/mdi/checkbox-blank-outline';
	import MdiCheckboxMarkedOutline from '~icons/mdi/checkbox-marked-outline';
	import MdiTable from '~icons/mdi/table';
	import MdiPaperAddOutline from '~icons/mdi/paper-add-outline';
	
	import type { TestInstance, TestItem } from '$lib/types/types.ts';

	let isPageLoading: boolean = $state(true);

	let activeTestInstance: TestInstance = $state({
		name: "",
		section: "",
		date: "",
		test_id: data.test_id,
		is_done_rendering: false,
	});

	onMount(async () => {
		try {
			const response = await fetch(
				`${apiBaseUrl}/api/test_instances/${data.test_id}`,
				{
					method: "GET",
					headers: {'Content-Type': 'application/json',},
					body: JSON.stringify({}),
				}
			);

			const result = await response.json();
			activeTestInstance.name = result.name;
			activeTestInstance.section = result.section;
			activeTestInstance.date = result.date;
			activeTestInstance.is_done_rendering = result.is_done_rendering;
		} catch (e) {
			// FIXME: revert to alert() once functional
			console.log("Failed to fetch test details:\n"+e);
		} finally {
			isPageLoading = false;
		}
	});

	setTimeout(() => {
		activeTestInstance.name = "Seatwork 1";
		activeTestInstance.section = "3-Rizal";
		activeTestInstance.date = "2025-11-11T20:17:46.384Z";
		activeTestInstance.is_done_rendering = false;

		isPageLoading = false;
	}, 500);
		
</script>

<div class="space-y-8">
	<!-- <TestInstanceTopbar
				{...activeTestInstance} name={isPageLoading ? "Loading..." : activeTestInstance.name} date={new Date(activeTestInstance.date).toLocaleDateString()} /> -->
	<div class="bg-sidebar px-4 py-8 border-b border-sidebar-border space-y-4">
		<span class="flex flex-row items-center gap-4">
			<a class="size-8" href="/">
				<MdiArrowBack class="size-full" />
			</a>
			<h1>{ activeTestInstance.name }</h1>
			{#if activeTestInstance.is_done_rendering}
				<MdiCheckboxMarkedOutline class="size-6 text-muted-foreground"/>
			{:else}
				<MdiCheckboxBlankOutline class="size-6 text-muted-foreground" />
			{/if}
		</span>
		<div id="change-instance-details">
			<h3>
				TestID:
				<span class="font-mono text-sm">{ activeTestInstance.test_id }</span>
			</h3>
			<h3>Section: { activeTestInstance.section }</h3>
			<h3>Date: { new Date(activeTestInstance.date).toLocaleString() }</h3>
		</div>
		<span id="thisOne" class="flex items-center gap-2 justify-end">
			<a class="button-primary" href={`/${data.test_id}/items`}>
				Items
			</a>
			<a class="button-primary" href={`/${data.test_id}/papers`}>
				Papers
			</a>
			<button class="button-primary">
				<MdiTable class="size-6" />
			</button>
			<button class="button-primary">
				<MdiPaperAddOutline class="size-6"/>
			</button>
		</span>
	</div>
	{@render children()}
</div>