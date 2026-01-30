<script lang="ts">
	let {
		name,
		section,
		date,
		test_id,
		is_done_rendering,
	} = $props();

	import MdiCheckboxBlankOutline from '~icons/mdi/checkbox-blank-outline';
	import MdiCheckboxMarkedOutline from '~icons/mdi/checkbox-marked-outline';

</script>

<a href={`/${test_id}/items`} class="flex flex-col rounded card px-3 py-1.5">
	<span class="flex flex-row items-center gap-1">
		{#if is_done_rendering}
			<MdiCheckboxMarkedOutline/>
		{:else}
			<MdiCheckboxBlankOutline/>
		{/if}
		<h3>{ name }</h3>
	</span>
	<h4>{ section } &centerdot; { new Date(date).toLocaleDateString() }</h4>
</a>
