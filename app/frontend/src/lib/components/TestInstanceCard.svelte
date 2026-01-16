<script lang="ts">
	export let test_name: string;
	export let section: string;
	export let date: string;
	export let is_done_rendering: boolean;

	const testid: string = test_name + section;
</script>

<div class="rounded card border border-border p-1">
	{#if is_done_rendering}
		<!-- TODO: add done rendering logo -->
	{:else}
		<!-- TODO: add not done rendering logo -->
	{/if}
	<a href={testid}>
		<h3>{test_name}</h3>
		<h4>{section} || {date}</h4>
	</a>
</div>
