<script lang="ts">
  const { data, children } = $props();
  import IconBack from "~icons/mdi/arrow-left";
  import IconHome from '~icons/mdi/home';

  const sectionId = Number(data.section_id);
  const sectionName = data.sections.find(
    (section) => section.section_id === sectionId
  )!.section_name;
</script>


<nav class="bg-sidebar text-sidebar-foreground
            shadow shadow-sidebar-border
            w-full flex flex-row justify-between p-4 pt-6">
  <a href="/instances"
      class="bg-white h-fit shadow-sm rounded-full">
    <IconHome class="size-8 text-primary foregroundize" />
  </a>
  <span class="*:leading-5 *:text-center">
    <h1>{sectionName}</h1>
    <h6>Class list</h6>
  </span>
  <a href="/sections">
    <IconBack class="size-8"/>
  </a>
</nav>

<span class="-my-4"></span>

{@render children()}

