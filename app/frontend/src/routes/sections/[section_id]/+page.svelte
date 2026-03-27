<script lang="ts">
	const { data } = $props();

	import MdiEdit from "~icons/mdi/edit";
  import IconBack from "~icons/mdi/arrow-left";

	import type { Student } from '$lib/index.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.ts';
	import Card from "$lib/components/Card.svelte";
	import AddNewStudent from './AddNewStudent.svelte';
	import EditStudent from './EditStudent.svelte';

	let students: Student[] = $derived(data.students);

  let isAddDialogOpen: boolean = $state(false);
</script>


<div class="container">
	<span class="flex flex-row space-x-4 mb-4">
		<a href="/sections">
			<IconBack class="size-8"/>
		</a>
    <span class="*:leading-5.5">
      <h1 class="align-left">Class List</h1>
      <h6>SECTION_ID {data.section_id}</h6>
    </span>
	</span>
  <div class="space-y-2">
    <Dialog.Root bind:open={isAddDialogOpen}>
      <Dialog.Trigger class="button-outline font-medium items-center w-full">
        (+) Add new student
      </Dialog.Trigger>
      <AddNewStudent bind:isAddDialogOpen
                      section_id={data.section_id}/>
    </Dialog.Root>

    {#each students as student}
      <Card>
        <span class="flex flex-row items-center justify-between">
          <h3>{student.name}</h3>
          <!-- TODO: add dialog bindable; close upon operation done. something similar to add new student-->
          <Dialog.Root>
            <Dialog.Trigger class="button-secondary">
              <MdiEdit class="size-4"/>
            </Dialog.Trigger>
            <EditStudent bind:section_id={data.section_id}
                          student_no={student.student_no}
                          bind:name={student.name}/>
          </Dialog.Root>
        </span>
        <h5>{student.student_no}</h5>
      </Card>
    {/each}
  </div>
</div>
