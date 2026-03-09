<script lang="ts">
  let { toggle = $bindable(), onDelete, size = 6 } = $props();
  import IconDelete from "~icons/mdi/delete";
</script>

{#if !toggle}
  <button class="button-destructive"
          onclick={() => toggle = true}>
    <IconDelete class={`size-${size} mx-auto`} />
  </button>
{:else}
  <button class="button-destructive {size <= 5 ? "text-xs" : "text-sm"}"
          onclick={onDelete}>
    Confirm delete
  </button>
{/if}