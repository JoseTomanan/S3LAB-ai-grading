<script lang="ts">
  const { data, children } = $props();
  import IconBack from "~icons/mdi/arrow-left";

  const sectionId = Number(data.section_id);
  const sectionName = data.sections.find(
    (section) => section.section_id === sectionId
  )!.section_name;
</script>


<nav class="bg-sidebar text-sidebar-foreground
            shadow shadow-sidebar-border
            w-full flex flex-row justify-between p-4 pt-6">
  <span class="*:leading-6">
    <h1 class="align-left">
      {sectionName}
    </h1>
    <h5>Class list</h5>
  </span>
  <a href="/sections">
    <IconBack class="size-8"/>
  </a>
</nav>

<span class="-my-4"></span>

{@render children()}

