<script lang="ts">
	export let test_name: string;
	export let section: string;
	export let date: string;
	export let is_done_rendering: boolean;

	import Icon from '@iconify/svelte';

	const testid: string = section + "-" + test_name;
	const dateObj: Date = new Date(date);
</script>

<a href={testid} class="flex flex-col rounded card bg-card px-3 py-1.5">
	<span class="flex flex-row items-center gap-1">
		<Icon icon={is_done_rendering ? 'proicons:checkbox-checked' : 'proicons:checkbox-unchecked'}
				class="size-6"/>
		<h3>{test_name}</h3>
	</span>
	<h4>{section} &centerdot; {dateObj.toLocaleDateString()}</h4>
</a>
