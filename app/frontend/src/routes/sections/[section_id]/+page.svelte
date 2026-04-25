<script lang="ts">
  import type { Student } from '$lib/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import BottomSheet from '$lib/components/BottomSheet.svelte';
  import { Button } from '$lib/components/CustomButton/index.ts';
  import AddNewStudent from './AddNewStudent.svelte';
  import EditStudent from './EditStudent.svelte';
  import BulkAddStudents from './BulkAddStudents.svelte';
  import IconPlus from '~icons/mdi/plus';
  import IconBulk from '~icons/mdi/table-multiple';
  import IconDots from '~icons/mdi/dots-horizontal';

  const { data } = $props();

  const sectionId = $derived(Number(data.section_id));
  let students: Student[] = $derived(data.students);
  let showAdd = $state(false);
  let showBulk = $state(false);
</script>

<div class="p-4 flex flex-col gap-2.5">
  <!-- section header -->
  <div class="flex items-center justify-between">
    <div>
      <span class="font-heading text-[20px] font-extrabold text-foreground tracking-[-0.04em]">Students</span>
      <span class="ml-2 text-[12px] font-heading font-bold px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
        {students.length} enrolled
      </span>
    </div>
    <div class="flex gap-1.5">
      <button
        onclick={() => { showBulk = true; }}
        class="size-9 rounded-full bg-background border border-border flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-border transition-colors"
        title="Bulk add"
      >
        <IconBulk class="size-[18px]" />
      </button>
      <button
        onclick={() => { showAdd = true; }}
        class="size-9 rounded-full bg-primary flex items-center justify-center text-white hover:opacity-90 transition-opacity"
        title="Add student"
      >
        <IconPlus class="size-[22px]" />
      </button>
    </div>
  </div>

  <!-- student cards -->
  {#each students as student}
    <div class="bg-card rounded-xl shadow-sm px-4 py-3 flex items-center gap-3">
      <!-- avatar -->
      <div class="size-[34px] rounded-full bg-accent flex items-center justify-center shrink-0">
        <span class="font-heading text-[13px] font-extrabold text-primary-700 tracking-[-0.02em]">
          {student.name.split(' ').map((n: string) => n[0]).slice(0, 2).join('')}
        </span>
      </div>

      <div class="flex-1 min-w-0">
        <p class="text-[14px] font-medium text-foreground truncate opacity-100">{student.name}</p>
        <p class="text-[11px] font-mono text-foreground/55 mt-0.5 opacity-100">{student.student_no}</p>
      </div>

      <!-- edit (Dialog kept for destructive UX) -->
      <Dialog.Root>
        <Dialog.Trigger
          class="size-8 rounded-full flex items-center justify-center text-foreground/55 hover:text-foreground hover:bg-muted transition-colors"
        >
          <IconDots class="size-5" />
        </Dialog.Trigger>
        <EditStudent
          section_id={sectionId}
          student_no={student.student_no}
          name={student.name}
        />
      </Dialog.Root>
    </div>
  {/each}

  {#if students.length === 0}
    <p class="text-center text-muted-foreground text-sm py-8">No students yet.</p>
  {/if}

  <Button variant="add" onclick={() => { showAdd = true; }}>
    <IconPlus class="size-[18px]" />
    Add student
  </Button>
</div>

<!-- Add student sheet -->
<BottomSheet bind:open={showAdd} title="Add Student">
  {#snippet children()}
    <AddNewStudent section_id={sectionId} close={() => { showAdd = false; }} />
  {/snippet}
</BottomSheet>

<!-- Bulk add dialog (kept as Dialog — denser form) -->
<Dialog.Root bind:open={showBulk}>
  <Dialog.Content class="max-w-lg">
    <BulkAddStudents section_id={sectionId} close={() => { showBulk = false; }} />
  </Dialog.Content>
</Dialog.Root>
