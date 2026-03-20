<script lang="ts">
		import { onMount } from 'svelte';
	
	const { data } = $props();
	
	import MdiEdit from "~icons/mdi/edit";
  import IconBack from "~icons/mdi/arrow-left";

	import type { Student } from '$lib/index.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.ts';
	import AddNewStudent from './AddNewStudent.svelte';
	import EditStudent from './EditStudent.svelte';
	import { Skeleton } from '$lib/components/ui/skeleton/index.ts';
	
	let isPageLoading = $state(true);
	let students: Student[] = $state([]);
	
	onMount(async () => {
		try {
			const response = await fetch(
						`/api/sections/${data.section_id}`,
						{
							method: "GET",
							headers: {'Content-Type': 'application/json',},
						}
						);

			switch (response.status) {
				case 200:
					const result = await response.json();
					students = result;
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch (e) {
			alert("Failed to fetch section details, students:\nERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	}); 
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
    <Dialog.Root>
      <Dialog.Trigger class="button-outline font-medium items-center w-full">
        (+) Add new student
      </Dialog.Trigger>
      <AddNewStudent section_id={data.section_id}/>
    </Dialog.Root>
    {#if isPageLoading}
      <Skeleton class="grayscale h-16"/>
    {:else}
      {#each students as student}
      <div class="card">
        <span class="flex flex-row items-center justify-between">
          <h3>{student.name}</h3>
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
      </div>
      {/each}
    {/if}
  </div>
</div>
