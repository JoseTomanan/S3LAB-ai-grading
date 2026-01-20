<script lang="ts">
	export let test_name: string;
	export let section: string;
	export let date: string;
	export let is_done_rendering: boolean;

	import MdiCheckboxBlank from '~icons/mdi/checkbox-blank';
	import MdiCheckboxMarked from '~icons/mdi/checkbox-marked';

	const testid: string = section + "-" + test_name;
</script>

<a href={testid + "/questions"} class="flex flex-col rounded card bg-card px-3 py-1.5">
	<span class="flex flex-row items-center gap-1">
		{#if is_done_rendering}
			<MdiCheckboxMarked/>
		{:else}
			<MdiCheckboxBlank/>
		{/if}
		<h3>{ test_name }</h3>
	</span>
	<h4>{ section } &centerdot; { new Date(date).toLocaleDateString() }</h4>
</a>
