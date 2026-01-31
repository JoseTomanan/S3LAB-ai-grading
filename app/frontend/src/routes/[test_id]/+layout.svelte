<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	let { data, children }: {data: LayoutData, children: Snippet} = $props();

	import type { LayoutData } from './$types.d.ts';
	import type { Snippet } from 'svelte';
	import { onMount } from 'svelte';
	
	import type { TestInstance, TestItem } from '$lib/types/types.ts';
	import TestInstanceTopbar from '$lib/components/TestInstanceTopbar.svelte';

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
		activeTestInstance = {
			name: "Seatwork 1",
			section: "3-Rizal",
			date: "2025-11-11T20:17:46.384Z",
			test_id: "Seatwork-1_3-Rizal",
			is_done_rendering: false
		};

		isPageLoading = false;
	}, 500);
		
</script>

<div class="space-y-8">
	<TestInstanceTopbar
				{...activeTestInstance} name={isPageLoading ? "Loading..." : activeTestInstance.name} date={new Date(activeTestInstance.date).toLocaleDateString()} />
	{@render children()}
</div>