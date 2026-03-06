<script lang="ts">
  let { filename, onchange } = $props();

	import { Input } from '$lib/components/ui/input/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import IconCheck from "~icons/mdi/check";

  let newFilename: string = $state(filename);
</script>


<Dialog.Content>
  <Dialog.Description>Change associated student number:</Dialog.Description>
  <div class="flex flex-row gap-x-1">
    <Input type="text"
            bind:value={newFilename}
            onblur={() => onchange(newFilename)}
            placeholder="Student no..."
        />
    <Dialog.Close class="button-secondary">
      <IconCheck />
    </Dialog.Close>
  </div>
</Dialog.Content>