<script lang="ts">
	export let name: string;
	export let section: string;
	export let date: string;
  export let test_id: string;
	export let is_done_rendering: boolean;

	import MdiCheckboxBlankOutline from '~icons/mdi/checkbox-blank-outline';
	import MdiCheckboxMarkedOutline from '~icons/mdi/checkbox-marked-outline';

</script>

<a href={`/${test_id}/questions`} class="flex flex-col rounded card px-3 py-1.5">
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
