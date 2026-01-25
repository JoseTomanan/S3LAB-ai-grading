<script lang="ts">
	export let test_name: string;
	export let section: string;
	export let date: string;
	export let is_done_rendering: boolean;

	import MdiCheckboxBlankOutline from '~icons/mdi/checkbox-blank-outline';
	import MdiCheckboxMarkedOutline from '~icons/mdi/checkbox-marked-outline';

	const testid: string = section + "-" + test_name;
	const dateObj: Date = new Date(date);
</script>

<a href={`/${testid}/questions`}
			class="flex flex-col rounded card bg-card px-4 py-2">
	<span class="flex flex-row items-center gap-2">
		{#if is_done_rendering}
			<MdiCheckboxMarkedOutline/>
		{:else}
			<MdiCheckboxBlankOutline/>
		{/if}
		<h3>{test_name}</h3>
	</span>
	<h4>{section} &centerdot; {dateObj.toLocaleDateString()}</h4>
</a>
