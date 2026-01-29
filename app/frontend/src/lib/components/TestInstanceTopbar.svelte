<script lang="ts">
	export let name: string;
	export let section: string;
	export let date: string;
	export let test_id: string;
	export let is_done_rendering: boolean;

	export let countShortFormItems: number;
	export let countProbSolItems: number;

	import MdiArrowBack from '~icons/mdi/arrow-back';
	import MdiCheckboxBlankOutline from '~icons/mdi/checkbox-blank-outline';
	import MdiCheckboxMarkedOutline from '~icons/mdi/checkbox-marked-outline';
	import MdiTable from '~icons/mdi/table';
	import MdiUpload from '~icons/mdi/upload';

	const dateObj: Date = new Date(date);
</script>

<div class="bg-sidebar p-4 border-b border-sidebar-border space-y-4">
	<span class="flex flex-row items-center gap-4">
		<a class="size-8" href="/">
			<MdiArrowBack class="size-full" />
		</a>
		<h1>{ name }</h1>
		{#if is_done_rendering}
			<MdiCheckboxMarkedOutline class="size-6 text-muted-foreground"/>
		{:else}
			<MdiCheckboxBlankOutline class="size-6 text-muted-foreground" />
		{/if}
	</span>
	<div id="change-instance-details">
		<h3>TEST_ID: { test_id }</h3>
		<h3>Section: { section }</h3>
		<h3>Date: { dateObj.toLocaleDateString() }</h3>
		<h3>No. of Short form: { countShortFormItems }</h3>
		<h3>No. of Problem solving: { countProbSolItems }</h3>
	</div>
	<span id="thisOne" class="flex items-center gap-2 justify-end">
		<button class="button-primary">Open class list</button>
		<button class="button-primary">
			<MdiTable class="size-6" />
		</button>
		<button class="button-primary">
			<MdiUpload class="size-6"/>
		</button>
	</span>
</div>