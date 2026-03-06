<script lang="ts">
  let { filename = $bindable() } = $props();

	import { Input } from './ui/input/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
</script>


<Dialog.Content>
  <Dialog.Description>Rename file (please keep the file extension):</Dialog.Description>
  <Input type="text"
          bind:value={filename}
          placeholder="Student no..." />
</Dialog.Content>