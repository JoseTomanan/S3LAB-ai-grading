<script>
	import MdiArrowForward from '~icons/mdi/arrow-forward';
	import MdiArrowBack from '~icons/mdi/arrow-back';
	import { Button } from '$lib/components/ui/button/index.ts';
	
	let { rows = [], perPage = 10, trimmedRows = $bindable() } = $props();

	let totalRows = $derived(rows.length);
	let currentPage = $state(0);
	let totalPages = $derived(Math.ceil(totalRows / perPage));
	let start = $derived(currentPage * perPage);
	let end = $derived(currentPage === totalPages - 1 ? totalRows - 1 : start + perPage - 1);

	$effect(() => {
		trimmedRows = rows.slice(start, end + 1);
	});

	$effect(() => {
		if (totalRows > 0) currentPage = 0;
	});
</script>

{#if totalRows && totalRows > perPage}
	<div class='pagination'>
		<Button variant="secondary"
					onclick={() => currentPage -= 1}
					disabled={currentPage === 0}
					aria-label="left arrow icon"
					aria-describedby="prev">
			<MdiArrowBack />
		</Button>
		<span id='prev' class='sr-only'>Load previous {perPage} rows</span>
		<p>{start + 1} - {end + 1} of {totalRows}</p>
		<Button variant="secondary"
					onclick={() => currentPage += 1}
					disabled={currentPage === totalPages - 1}
					aria-label="right arrow icon"
					aria-describedby="next">
			<MdiArrowForward />
		</Button>
		<span id='next' class='sr-only'>Load next {perPage} rows</span>
	</div>
{/if}


<style>
	.sr-only {
		position: absolute;
		clip: rect(1px, 1px, 1px, 1px);
		padding: 0;
		border: 0;
		height: 1px;
		width: 1px;
		overflow: hidden;
	}
	
	.pagination {
			display: flex;
			align-items: center;
			justify-content: center;
			pointer-events: all;
	}

	.pagination p {
			margin: 0 1rem;
	}


</style>