<script lang="ts">
	const { data } = $props();

	import MdiEdit from "~icons/mdi/edit";
  import IconPlus from "~icons/mdi/plus";

	import type { Student } from '$lib/index.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { buttonVariants } from '$lib/components/CustomButton/index.ts';
  import { cn } from '$lib/utils.ts';
	import Card from "$lib/components/Card.svelte";
	import AddNewStudent from './AddNewStudent.svelte';
	import EditStudent from './EditStudent.svelte';

	let students: Student[] = $derived(data.students);

  let isAddDialogOpen: boolean = $state(false);
</script>


<div class="container space-y-3">
  <h1>Class list</h1>
  
  <div class="space-y-2">
    <Dialog.Root bind:open={isAddDialogOpen}>
      <Dialog.Trigger class={cn(buttonVariants({ variant: 'outline' }), 'flex flex-row gap-x-1 justify-center items-center w-full font-medium')}>
        <IconPlus />
        Add new student
      </Dialog.Trigger>
      <AddNewStudent bind:isAddDialogOpen
                      section_id={data.section_id}/>
    </Dialog.Root>

    {#each students as student}
      <Card class="flex flex-row items-center justify-between">
        <div class="space-x-1">
          <b class="font-medium tracking-tighter">{student.student_no}</b>
          <span>{student.name}</span>
        </div>
        <!-- TODO: add dialog bindable; close upon operation done. something similar to add new student-->
        <Dialog.Root>
          <Dialog.Trigger class={buttonVariants({ variant: 'secondary' })}>
            <MdiEdit class="size-4"/>
          </Dialog.Trigger>
          <EditStudent bind:section_id={data.section_id}
                        student_no={student.student_no}
                        bind:name={student.name}/>
        </Dialog.Root>
      </Card>
    {/each}
  </div>
</div>
