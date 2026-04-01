<script lang="ts">
  let { toggle = $bindable(), onDelete, size = 6 } = $props();
  import { Button } from '$lib/components/ui/button/index.ts';
  import IconDelete from "~icons/mdi/delete";
</script>

{#if !toggle}
  <Button variant="destructive" onclick={() => toggle = true}>
    <IconDelete class={`size-${size} mx-auto`} />
  </Button>
{:else}
  <Button variant="destructive" class={size <= 5 ? 'text-xs' : 'text-sm'} onclick={onDelete}>
    Confirm delete
  </Button>
{/if}