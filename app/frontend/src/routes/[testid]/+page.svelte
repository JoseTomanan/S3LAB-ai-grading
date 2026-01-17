<script lang="ts">
	// export let test_name: string;
	// export let section: string;
	// export let date: string;
	// export let is_done_rendering: boolean;
	
	import Icon from '@iconify/svelte';

	$: test_name = "Seatwork 1";
	$: section = "3-Rizal";
	$: string = "";
	$: is_done_rendering = false;

</script>

<div class="px-2 py-4">
	<div class="bg-sidebar p-4 border-b border-sidebar-border space-y-2">
		<span class="flex flex-row items-center gap-2">
			<span class="size-8">
				<Icon icon="proicons:arrow-left" class="size-full"/>
			</span>
			<h1>{test_name}</h1>
		</span>
		<div id="change-instance-details">
			<h3>Section: {section}</h3>
			<h3>No. of Short-form: ...</h3>
			<h3>No. of Problem solving: ...</h3>
		</div>
		<span>
			<!-- TODO : open class list button -->
			<!-- TODO : open spreadsheet button -->
			<!-- TODO : batch upload button -->
		</span>
	</div>
	<div>
	
	</div>
</div>