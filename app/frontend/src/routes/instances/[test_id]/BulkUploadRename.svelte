<script lang="ts">
	import { untrack } from 'svelte';
  let { filename, onchange } = $props();

	import { Input } from '$lib/components/ui/input/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { buttonVariants } from '$lib/components/ui/button/index.ts';
  import IconCheck from "~icons/mdi/check";
	import { Label } from '$lib/components/ui/label/index.ts';

  let newFilename: string = $state(untrack(() => filename));
</script>


<Dialog.Content>
  <Label for="filename">Change associated student number:</Label>
  <div class="flex flex-row gap-x-1">
    <Input id="filename"
            type="text"
            bind:value={newFilename}
            onblur={() => onchange(newFilename)}
            placeholder="Student no..."
        />
    <Dialog.Close class={buttonVariants({ variant: 'secondary' })}>
      <IconCheck />
    </Dialog.Close>
  </div>
</Dialog.Content>