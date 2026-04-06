<script lang="ts">
  import type { Snippet } from 'svelte';
  import * as Dialog from '$lib/components/ui/dialog';
  import IconDelete from "~icons/mdi/delete";
  import { cn } from '$lib/utils';
  import Button, { buttonVariants } from './CustomButton/button.svelte';

  type Props = {
    children?: Snippet;
    onDelete: () => void;
  };
  let { children, onDelete }: Props = $props();

  let open = $state(false);
</script>

<Dialog.Root bind:open>
  <Dialog.Trigger class={cn(buttonVariants({ variant: 'destructive' }), "")}>
    {#if children}
      {@render children()}
    {:else}
      <IconDelete class="size-full"/>
    {/if}
  </Dialog.Trigger>
  <Dialog.Content>
    <Dialog.Description>
      Are you sure? This action cannot be undone.
    </Dialog.Description>
    <Dialog.Footer>
      <Button variant="destructive"
              onclick={() => {onDelete(); open=false;}}>
        Confirm delete
      </Button>
      <Dialog.Close class={buttonVariants({ variant: 'outline' })}>
        Cancel
      </Dialog.Close>
    </Dialog.Footer>
  </Dialog.Content>
</Dialog.Root>
