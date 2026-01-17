<script lang="ts">
	// export let test_name: string;
	// export let section: string;
	// export let date: string;
	// export let is_done_rendering: boolean;

	import MdiArrowBack from '~icons/mdi/arrow-back';
	import MdiTable from '~icons/mdi/table';
	import MdiUpload from '~icons/mdi/upload';

	$: test_name = "Seatwork 1";
	$: section = "3-Rizal";
	$: date = "2025-11-11T20:17:46.384Z";
	$: is_done_rendering = false;

	$: countShortFormItems = 0;
	$: countProbSolItems = 0;

</script>

<div class="px-2 py-4">
	<div class="bg-sidebar p-4 border-b border-sidebar-border space-y-2">
		<span class="flex flex-row items-center gap-2">
			<a class="size-8" href=".">
				<MdiArrowBack class="size-full" />
			</a>
			<h1>{test_name}</h1>
		</span>
		<div id="change-instance-details">
			<h3>Section: {section}</h3>
			<h3>No. of Short-form: {countShortFormItems}</h3>
			<h3>No. of Problem solving: {countProbSolItems}</h3>
		</div>
		<span id="thisOne" class="flex items-center gap-2 justify-end">
			<button>Open class list</button>
			<button><MdiTable class="size-6"/></button>
			<button><MdiUpload class="size-6"/></button>
		</span>
	</div>
	<div>
	
	</div>
</div>