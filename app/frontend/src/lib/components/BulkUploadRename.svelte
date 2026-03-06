<script lang="ts">
  let { filename = $bindable() } = $props();

	import { Input } from './ui/input/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';

  import IconCheck from "~icons/mdi/check";
</script>


<Dialog.Content showCloseButton={false}>
  <Dialog.Description>Indicate student no. (keep the file extension):</Dialog.Description>
  <div class="flex flex-row gap-x-1">
    <Input type="text"
            bind:value={filename}
            placeholder="Student no..."
        />
    <Dialog.Close class="button-secondary">
      <IconCheck />
    </Dialog.Close>
  </div>
</Dialog.Content>