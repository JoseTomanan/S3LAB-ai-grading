<script lang="ts">
  import { API_BASE_URL } from '$lib/constants.ts';

  const { test_id } = $props();

  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { Button } from '$lib/components/ui/button/index.ts';
	import { Input } from '$lib/components/ui/input/index.ts';

  async function bulkUpload() {
    // TODO: function
  }
</script>


<Dialog.Content>
  <Dialog.Header>
    <Dialog.Title>Upload images</Dialog.Title>
  </Dialog.Header>
  <!-- TODO: everything else -->
</Dialog.Content>