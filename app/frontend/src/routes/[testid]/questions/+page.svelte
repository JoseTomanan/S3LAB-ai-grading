<script lang="ts">
	export let test_name: string;
	export let section: string;
	export let date: string;
	export let is_done_rendering: boolean;

	import MdiEdit from '~icons/mdi/edit';

	import type { TestItem } from '$lib/types/types.ts';
	import TestDetails from '$lib/components/TestInstanceTopbar.svelte';
	
	let allItems: TestItem[] = [];
	let shortFormItems: TestItem[];
	let probSolItems: TestItem[];
	
	function retrieveItems() {
		// TODO: function that retrieves from API call and populate allQuestions
	}
	
	$: shortFormItems = allItems.filter(item => !item.is_problem_solving);
	$: probSolItems = allItems.filter(item => item.is_problem_solving);
	
	// temporary!
	test_name = "Seatwork 1";
	section = "3-Rizal";
	date = "2025-11-11T20:17:46.384Z";
	is_done_rendering = false;

	// temporary!
	allItems = [
		{number: 1, question: "David and Goliath divide a pie in half. If they were to divide it evenly, how many should each one get?", is_problem_solving: false, expected_answer_rubric_question: ""},
		{number: 2, question: "Three people are to share a pie evenly. Using a circle, illustrate how this pie will be sliced.", is_problem_solving: false, expected_answer_rubric_question: ""},
	];

</script>

<div class="py-4 space-y-8">
	<TestDetails {test_name} {section} {date} {is_done_rendering}
				countShortFormItems={shortFormItems.length} countProbSolItems={probSolItems.length}/>
	<div class="px-4 space-y-4">
		<div class="card p-2">
			<span class="flex flex-row items-center w-full justify-between">
				<h3>Short Form Items</h3>
				<button on:click={() => {}}>
					<MdiEdit/>
				</button>
			</span>
			<div class="ml-4">
				{#each shortFormItems as item}
					<p class="truncate text-ellipsis">
						{item.number}: {item.question}
					</p>
				{/each}
			</div>
		</div>
		<div class="card p-2">
			<span class="flex flex-row items-center w-full justify-between">
				<h3>Problem Solving Items</h3>
				<button on:click={() => {}}>
					<MdiEdit/>
				</button>
			</span>
			<div class="ml-4">
				{#each probSolItems as item}
					<p class="truncate text-ellipsis">
						{item.number}: {item.question}
					</p>
				{/each}
			</div>
		</div>
	</div>
</div>