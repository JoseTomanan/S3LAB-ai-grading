<script lang="ts">
	export let test_name: string;
	export let section: string;
	export let date: string;
	export let is_done_rendering: boolean;

	import Icon from '@iconify/svelte';

	const testid: string = test_name + section;
	const dateObj: Date = new Date(date);
</script>

<div class="flex gap-2 rounded card border border-border px-3 py-1.5">
	<a href={testid}>
		<h3 class="flex flex-row items-center gap-1">
			{#if is_done_rendering}
				<Icon icon="proicons:checkbox-checked" class="size-6"/>
			{:else}
				<Icon icon="proicons:checkbox-unchecked" class="size-6"/>
			{/if}
			{test_name}
		</h3>
		<h4>{section} &centerdot; {dateObj.toLocaleDateString()}</h4>
	</a>
</div>
