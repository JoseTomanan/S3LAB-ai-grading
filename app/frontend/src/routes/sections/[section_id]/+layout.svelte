<script lang="ts">
  import type { Snippet } from 'svelte';
  import TopBar from '$lib/components/TopBar.svelte';

  const { data, children }: { data: any; children: Snippet } = $props();

  const sectionName = $derived(
    data.sections.find((s: any) => s.section_id === Number(data.section_id))?.section_name ?? ''
  );
</script>

<div class="flex flex-col min-h-full">
  <TopBar title={sectionName} back="/sections" />
  {@render children()}
</div>
