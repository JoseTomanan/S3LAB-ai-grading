<script lang="ts">
	// export let test_name: string;
	// export let section: string;
	// export let date: string;
	// export let is_done_rendering: boolean;

	const test_name: string = "Seatwork 1";
	const section: string = "3-Rizal";
	const date: string = "";
	const is_done_rendering: boolean = false;

	import Icon from '@iconify/svelte';
</script>

<div class="px-2 py-4">
	<div class="bg-secondary p-4 border-b border-border">
		<span class="flex flex-row items-center gap-2">
			<Icon icon="proicons:arrow-left" class="size-8"/>
			<h1>{test_name}</h1>
		</span>
		<h3>Section: {section}</h3>
		<h3>No. of Short-form: ...</h3>
		<h3>No. of Problem Solving: ...</h3>
	</div>
	<div>
	
	</div>
</div>