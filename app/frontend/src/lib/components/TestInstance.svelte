<script lang="ts">
	export let test_name: string;
	export let section: string;
	export let date: string;
	export let is_done_rendering: string;
</script>

<div class="rounded card">
	{#if is_done_rendering}
		<!-- TODO: add done rendering logo -->
	{:else}
		<!-- TODO: add not done rendering logo -->
	{/if}
	<div>
		<h3><a href="_blank">{test_name}</a></h3>
		<h4>{section} || {date}</h4>
	</div>
</div>
