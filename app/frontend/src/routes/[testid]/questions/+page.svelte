<script lang="ts">
	export let test_name: string;
	export let section: string;
	export let date: string;
	export let is_done_rendering: boolean;

	import MdiEdit from '~icons/mdi/edit';

	import type { TestQuestion } from '$lib/types/types.ts';
	import TestDetails from '$lib/components/TestInstanceTopbar.svelte';

	// temporary!
	test_name = "Seatwork 1";
	section = "3-Rizal";
	date = "2025-11-11T20:17:46.384Z";
	is_done_rendering = false;
	
	let allItems: TestQuestion[] = [];
	let shortFormItems: String[];
	let probSolItems: String[];

	function retrieveItems() {
		// TODO: function that retrieves from API call and populate allQuestions
	}

	// temporary
	allItems = [];

	$: shortFormItems = allItems.filter(item => !item.is_problem_solving).map(item => item.question);
	$: probSolItems = allItems.filter(item => item.is_problem_solving).map(item => item.question);

</script>

<div class="py-4 space-y-8">
	<TestDetails {test_name} {section} {date} {is_done_rendering}
				countShortFormItems={shortFormItems.length} countProbSolItems={probSolItems.length}/>
	<div class="px-4 space-y-4">
		{#each [{name: "Short form Items", items: shortFormItems}, {name: "Problem solving Items", items: probSolItems}] as listOfItems}
			<div class="p-2 bg-card ring">
				<span class="flex flex-row items-center w-full justify-between">
					<h3>{ listOfItems.name }</h3>
					<MdiEdit/>
				</span>
				{#each listOfItems.items as item}
					{ item }
				{/each}
			</div>
		{/each}
	</div>
</div>