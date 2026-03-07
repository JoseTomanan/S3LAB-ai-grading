<script lang="ts">
  let { filename, onchange } = $props();

	import { Input } from '$lib/components/ui/input/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import IconCheck from "~icons/mdi/check";
	import { Label } from '$lib/components/ui/label/index.ts';

  let newFilename: string = $state(filename);
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
    <Dialog.Close class="button-secondary">
      <IconCheck />
    </Dialog.Close>
  </div>
</Dialog.Content>