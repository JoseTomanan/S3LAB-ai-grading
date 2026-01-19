<script lang="ts">
	export let test_name: string;
	export let section: string;
	export let date: string;
	export let is_done_rendering: boolean;

	import MdiArrowBack from '~icons/mdi/arrow-back';
	import MdiEdit from '~icons/mdi/edit';
	import MdiTable from '~icons/mdi/table';
	import MdiUpload from '~icons/mdi/upload';

	import type { TestQuestion } from '$lib/types.ts';

	$: test_name = "Seatwork 1";
	$: section = "3-Rizal";
	$: date = "2025-11-11T20:17:46.384Z";
	$: is_done_rendering = false;
	
	let shortFormItems: TestQuestion[];
	let probSolItems: TestQuestion[];

	$: shortFormItems = [];
	$: probSolItems = [];

	function retrieveItems() {
		// TODO: function that retrieves from API call and populates respective arrays
	}

	// TODO: SIMPLIFY SHORTFORMITEMS TYPE SHIT

</script>

<div class="py-4 space-y-8">
	<div class="bg-sidebar p-4 border-b border-sidebar-border space-y-4">
		<span class="flex flex-row items-center gap-2">
			<a class="size-8" href=".">
				<MdiArrowBack class="size-full" />
			</a>
			<h1>{ test_name }</h1>
		</span>
		<div id="change-instance-details">
			<h3>Section: { section }</h3>
			<h3>No. of Short form: { shortFormItems.length }</h3>
			<h3>No. of Problem solving: { probSolItems.length }</h3>
		</div>
		<span id="thisOne" class="flex items-center gap-2 justify-end">
			<button class="button-primary">Open class list</button>
			<button class="button-primary"><MdiTable class="size-6"/></button>
			<button class="button-primary"><MdiUpload class="size-6"/></button>
		</span>
	</div>
	<div class="px-4 space-y-4">
		{#each [{name: "Short form Items", items: shortFormItems}, {name: "Problem solving Items", items: probSolItems}] as listOfItems}
			<div class="p-2 border border-border bg-card">
				<span class="flex flex-row items-center w-full justify-between">
					<h3>{ listOfItems.name }</h3>
					<MdiEdit/>
				</span>
				{#each listOfItems.items as item}
					{ item.question }
				{/each}
			</div>
		{/each}
	</div>
</div>