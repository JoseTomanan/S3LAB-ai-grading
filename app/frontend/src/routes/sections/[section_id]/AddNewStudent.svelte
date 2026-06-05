<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
  import Button from '$lib/components/CustomButton/button.svelte';
  import { Input } from '$lib/components/ui/input/index.ts';
  import { Label } from '$lib/components/ui/label/index.ts';

  interface Props { section_id: number | undefined; close: () => void }
  let { section_id, close }: Props = $props();

  let isLoading = $state(false);
  let formName = $state('');
  let formStudentNo = $state('');

  async function addNewStudent() {
    if (!formStudentNo || !formName) {
      toast('Please fill in all fields');
      return;
    }
    isLoading = true;
    try {
      await api(`${API_URL}/api/students`, {
        method: 'POST',
        body: JSON.stringify({ student_no: formStudentNo, name: formName, section_id }),
      });
      toast.success('Student added.');
      await invalidateAll();
      close();
    } catch (e) {
      toast.error(
        e instanceof ApiError
          ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
          : 'Failed to add student:\n' + e
      );
    } finally {
      isLoading = false;
    }
  }
</script>

<div class="flex flex-col gap-4">
  <div>
    <Label for="new_student_no" class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">
      Student number
    </Label>
    <Input id="new_student_no" placeholder="e.g. 2024-00421" bind:value={formStudentNo} />
    <p class="text-[11px] text-muted-foreground mt-1">Student number cannot be changed after creation.</p>
  </div>

  <div>
    <Label for="new_name" class="text-[11px] font-semibold uppercase tracking-[0.04em] text-muted-foreground mb-1.5 block">
      Full name
    </Label>
    <Input id="new_name" placeholder="e.g. Juan dela Cruz" bind:value={formName} />
  </div>

  <div class="flex gap-2 mt-1">
    <Button variant="ghost" class="flex-1" onclick={close}>Cancel</Button>
    <Button variant="primary" class="flex-1" disabled={isLoading} onclick={addNewStudent}>
      {isLoading ? 'Adding…' : 'Save'}
    </Button>
  </div>
</div>
