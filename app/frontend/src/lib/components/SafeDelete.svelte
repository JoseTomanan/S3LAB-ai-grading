<script lang="ts">
  let { toggle = $bindable(), onDelete } = $props();
  import IconDelete from "~icons/mdi/delete";
</script>

{#if !toggle}
  <button class="button-destructive"
          onclick={() => toggle = true}>
    <IconDelete class="size-6 mx-auto"/>
  </button>
{:else}
  <button class="button-destructive text-sm"
          onclick={onDelete}>
    Confirm delete
  </button>
{/if}